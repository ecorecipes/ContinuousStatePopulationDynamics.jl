# Resilience operators from continuous-state problems: methods of `ResilienceCore.linearise`.
#
# Two routes, chosen by whether a state `u` is given:
# - without `u`, the problem must be linear (affine) in its state, and its operator is returned:
#   a `Generator` (with mesh weights), the Jacobian at the equilibrium of an affine problem, or a
#   `PeriodicOperator` over one period of a time-varying problem;
# - with `u`, the problem is linearised at `u` through its SciML `ODEProblem`, which needs `u` to
#   be an equilibrium and ForwardDiff to be loaded.

"""
    linearise(prob::ContinuousIPMProblem, u = nothing; p = prob.p, t = first(prob.tspan),
              period = nothing, steps = 100, weights = ones(length(prob.u0)),
              tol = 1e-8, check = true)

The ResilienceCore operator of a continuous-time generator problem `du/dt = G u + b`.

Without `u`:
- a generator that does not depend on the state (a matrix, or a callable `(p, t)` / `(t, p)`)
  and no input term gives `Generator(G(p, t); weights)`;
- with a constant input term `b`, the equilibrium `u* = -G⁻¹ b` (G must be stable) and the
  `ContinuousJacobian(G, u*)` of the return to it;
- with `period = T`, a time-varying generator gives `periodic_generator`'s `PeriodicOperator`
  over one period starting at `t`, with the generator held at the midpoints of `steps` equal
  sub-intervals. Its metrics are per sub-step `period / steps` (see `periodic_generator`). A
  periodic input term has no single equilibrium and is not supported.

With `u`, the Jacobian at the equilibrium `u` of `to_ode_problem(prob)`: the route for a
generator or input term that depends on the state (callables `(u, p, t)` or `(u, t, p)`,
which are always treated as state-dependent). `u` must satisfy `G(u) u + b(u) ≈ 0`, unless
`check = false`, and ForwardDiff must be loaded.

**Weights.** The default weights are one: the demographic realisation of this problem
(`to_demographic_reactions`) treats the state as counts per mesh bin. If your state holds
densities at mesh points, pass `weights = mesh_weights(prob.domain)`. On a uniform mesh a common
weight changes no metric; it matters only on non-uniform weights.

**Normalisation.** With `normalize = true` the problem follows proportions, `du/dt = G u -
(1ᵀG u / 1ᵀu) u`, which is not linear. Its operator is still the `Generator(G)`: the
proportions approach the stable distribution at the rate given by the gap between the real
parts of the two leading eigenvalues of `G`, and the growth rate of `G` is the rate that the
normalisation removes. With an input term, or a state-dependent generator, pass `u`.

`DelayIPMProblem`s and demographic realisations have no single operator and throw an error.
"""
function ResilienceCore.linearise(prob::ContinuousIPMProblem, u = nothing; p = prob.p,
        t = first(prob.tspan), period = nothing, steps::Integer = 100,
        weights::AbstractVector = ones(length(prob.u0)), tol::Real = 1e-8, check::Bool = true)
    u === nothing || return _linearise_at(prob, u, p, t, period, tol, check)
    n = length(prob.u0)
    gen = _resolve_generator(prob.generator, prob.u0, p, t)
    src = _resolve_source(prob.source, prob.u0, p, t)
    if gen.mode in (1, 2) || (src !== nothing && src.mode in (1, 2))
        throw(ArgumentError("the generator or the input term of this problem depends on the state, so it " *
                            "has no single operator: pass an equilibrium `u` to linearise it there"))
    end
    G(s) = Matrix{Float64}(_materialize_generator_matrix(_call(gen, prob.u0, p, s), n))
    b(s) = src === nothing ? zeros(n) : Vector{Float64}(_call(src, prob.u0, p, s))
    if period !== nothing
        src === nothing ||
            throw(ArgumentError("a periodic problem with an input term has no single equilibrium; " *
                                "`period` needs a problem without `source`"))
        return periodic_generator(s -> G(t + s), period; steps = steps, weights = weights)
    end
    b0 = b(t)
    iszero(b0) && return Generator(G(t); weights = weights)
    prob.normalize &&
        throw(ArgumentError("a normalised problem with an input term is not linear: pass an equilibrium `u`"))
    G0 = G(t)
    return ContinuousJacobian(G0, affine_equilibrium(G0, b0))
end

"""
    linearise(prob::PSPMIPMProblem, u = nothing; p = prob.p, t = first(prob.tspan),
              period = nothing, steps = 100, weights = mesh_weights(prob.domain),
              tol = 1e-8, check = true)

The ResilienceCore operator of a physiologically structured transport problem, discretised by
its `FixedMeshUpwind` scheme.

Without `u`, the problem must be affine in its population, `dn/dt = T n + b`: velocity,
mortality, source and boundary fluxes may depend on the trait, `p` and `t`, but not on the
population, and there is no auxiliary state. The transport matrix `T` is assembled exactly, by
applying the problem's right-hand side to the unit vectors, and checked against the right-hand
side on random states; a problem that fails the check is treated as state-dependent. Then:
- with no inflow or source (`b = 0`), `Generator(T; weights)`;
- otherwise, the equilibrium `n* = -T⁻¹ b` (T must be stable) and `ContinuousJacobian(T, n*)`;
- with `period = T`, the `PeriodicOperator` of `periodic_generator` over one period starting at
  `t` (only when `b = 0` throughout the period), whose metrics are per sub-step `period / steps`.

With `u = [n; aux]`, the Jacobian at that equilibrium of `to_ode_problem(prob)`: the route for
density dependence and for auxiliary state such as a resource. It needs ForwardDiff to be
loaded.

**Weights.** The upwind scheme evolves densities at the mesh midpoints (the flux difference is
divided by the bin width), so the default weights are `mesh_weights(prob.domain)`, the bin width
of each mesh point. A `ContinuousJacobian` carries no weights, so they are dropped on the affine
and state-dependent routes; on a uniform mesh with no auxiliary state this changes no metric,
but with auxiliary state the norm-based metrics (reactivity, amplification) treat a unit of
density and a unit of each auxiliary variable alike.

**Normalisation.** As for `ContinuousIPMProblem`: without an input term the operator is that of
the unnormalised problem; otherwise pass `u`.
"""
function ResilienceCore.linearise(prob::PSPMIPMProblem, u = nothing; p = prob.p,
        t = first(prob.tspan), period = nothing, steps::Integer = 100,
        weights::AbstractVector = mesh_weights(prob.domain), tol::Real = 1e-8, check::Bool = true)
    u === nothing || return _linearise_at(prob, u, p, t, period, tol, check)
    isempty(prob.aux0) ||
        throw(ArgumentError("this problem has auxiliary state, so it has no single operator: pass an " *
                            "equilibrium `u = [n; aux]` to linearise it there"))
    f = to_ode_problem(prob.normalize ? remake(prob; normalize = false) : prob).f
    n = length(prob.n0)
    T, b = _affine_parts(f, n, p, t)
    _isaffine(f, T, b, p, t, prob.n0) ||
        throw(ArgumentError("the right-hand side of this problem is not affine in the population (a field " *
                            "or boundary flux depends on it): pass an equilibrium `u` to linearise it there"))
    if period !== nothing
        part = s -> begin
            Ts, bs = _affine_parts(f, n, p, t + s)
            iszero(bs) ||
                throw(ArgumentError("a periodic problem with inflow or a source has no single equilibrium; " *
                                    "`period` needs zero source and inflow"))
            Ts
        end
        return periodic_generator(part, period; steps = steps, weights = weights)
    end
    iszero(b) && return Generator(T; weights = weights)
    prob.normalize &&
        throw(ArgumentError("a normalised problem with inflow or a source is not linear: pass an equilibrium `u`"))
    return ContinuousJacobian(T, affine_equilibrium(T, b))
end

function ResilienceCore.linearise(prob::DelayIPMProblem, args...; kwargs...)
    throw(ArgumentError("a delay problem has no single operator: its stability is set by a characteristic " *
                        "equation with infinitely many roots, not by the eigenvalues of its undelayed part. " *
                        "Approximate the delay by stages (a linear chain) and linearise that, or use " *
                        "simulation-based metrics"))
end

function ResilienceCore.linearise(sol::ContinuousDemographicSolution, args...; kwargs...)
    throw(ArgumentError("a demographic (stochastic) realisation has no operator: linearise the problem it " *
                        "was simulated from, or use trajectory-based metrics on the realisations"))
end

# The Jacobian at the equilibrium `u` of the SciML lowering (ResilienceCore's SciML route).
function _linearise_at(prob, u, p, t, period, tol, check)
    period === nothing ||
        throw(ArgumentError("`period` describes a linear time-varying problem; it cannot be combined with `u`"))
    return linearise(to_ode_problem(prob), collect(float.(u)); p = p, t = t, tol = tol, check = check)
end

# The affine parts of an in-place right-hand side `f!(du, u, p, t)` on `n` states:
# `b = f(0)` and the columns `T[:, j] = f(e_j) - b`.
function _affine_parts(f!, n::Int, p, t)
    du = zeros(n)
    x = zeros(n)
    f!(du, x, p, t)
    b = copy(du)
    T = zeros(n, n)
    for j in 1:n
        x[j] = 1.0
        f!(du, x, p, t)
        T[:, j] .= du .- b
        x[j] = 0.0
    end
    return T, b
end

# Whether `f!` agrees with `T x + b` on random states of the scale of `x0`.
function _isaffine(f!, T, b, p, t, x0; trials::Integer = 2)
    n = length(b)
    scale = max(1.0, maximum(abs, x0; init = 0.0))
    rng = Random.Xoshiro(20191217)
    du = zeros(n)
    for _ in 1:trials
        x = scale .* (0.5 .+ rand(rng, n))
        f!(du, x, p, t)
        expected = T * x .+ b
        norm(du .- expected, Inf) <= 1e-8 * (1 + norm(T, Inf) * norm(x, Inf) + norm(b, Inf)) || return false
    end
    return true
end
