"""
Population-structured transport problem types for continuous-state population
dynamics.
"""

"""
    AbstractTransportDiscretization

Supertype for numerical discretization strategies used to lower transport PDEs
to ODE systems.
"""
abstract type AbstractTransportDiscretization end

"""
    FixedMeshUpwind()

First-order upwind method-of-lines discretization on a fixed
`ContinuousDomain`.
"""
struct FixedMeshUpwind <: AbstractTransportDiscretization end

"""
    PSPMIPMProblem(structure, domain, n0, tspan;
                   velocity, mortality, boundary_lower, boundary_upper,
                   source, aux0, auxiliary_rhs, discretization, p, normalize)

Single-trait transport problem over a `ContinuousDomain`, plus optional coupled
auxiliary ODE state.

# Field arguments

The `velocity`, `mortality`, and `source` fields may be a `Number` (constant
over the domain), an `AbstractVector` of per-location values, or a callable.
**Callables are vectorized**: they are passed the entire mesh as a vector, not
one location at a time. `mortality` and `source` receive the mesh midpoints
(`meshpoints(domain)`, length `n`); `velocity` receives the bin edges
(`bounds(domain)`, length `n + 1`), or it may return `n` cell values that are
interpolated to faces. Supported callable signatures, tried in order, are
`(points, population, aux, p, t)`, `(points, population, p, t)`,
`(points, aux, p, t)`, `(points, p, t)`, `(points, t, p)`, and `(points)`.
Because a generic closure matches the first signature it has the right arity
for, write field functions against the whole-mesh-vector convention (e.g.
`z -> 0.2 .* z`), not against a single scalar location (`z -> 0.2 * z` happens
to work only because scalar multiplication broadcasts; `z -> z^2` would not).

`boundary_lower`/`boundary_upper` are scalar fluxes (a `Number` or a callable
returning one); they are used at an inflow edge, while outflow edges use the
upwinded interior value.
"""
struct PSPMIPMProblem{
        S<:AbstractIPMStructure,
        Dom<:ContinuousDomain,
        N,
        A,
        T<:Real,
        V,
        M,
        BL,
        BU,
        So,
        R,
        D<:AbstractTransportDiscretization,
        P} <: AbstractContinuousIPMProblem
    structure::S
    domain::Dom
    n0::N
    aux0::A
    tspan::Tuple{T, T}
    velocity::V
    mortality::M
    boundary_lower::BL
    boundary_upper::BU
    source::So
    auxiliary_rhs::R
    discretization::D
    p::P
    normalize::Bool
end

function _pspm_numeric_type(n0, aux0, tspan)
    aux_T = isempty(aux0) ? Float64 : eltype(aux0)
    return promote_type(
        Float64,
        eltype(n0),
        aux_T,
        typeof(float(tspan[1])),
        typeof(float(tspan[2])),
    )
end

function PSPMIPMProblem(structure::AbstractIPMStructure,
        domain::ContinuousDomain,
        n0,
        tspan;
        velocity,
        mortality = 0.0,
        boundary_lower = 0.0,
        boundary_upper = 0.0,
        source = 0.0,
        aux0 = Float64[],
        auxiliary_rhs = nothing,
        discretization::AbstractTransportDiscretization = FixedMeshUpwind(),
        p = nothing,
        normalize::Bool = false)
    n0_vec = collect(n0)
    isempty(n0_vec) && throw(ArgumentError("n0 must contain at least one state value"))
    _validate_continuous_state(domain, n0_vec)
    aux0_vec = collect(aux0)
    T = _pspm_numeric_type(n0_vec, aux0_vec, tspan)
    return PSPMIPMProblem(
        structure,
        domain,
        T.(n0_vec),
        T.(aux0_vec),
        (T(tspan[1]), T(tspan[2])),
        velocity,
        mortality,
        boundary_lower,
        boundary_upper,
        source,
        auxiliary_rhs,
        discretization,
        p,
        normalize,
    )
end

function PSPMIPMProblem(domain::ContinuousDomain, n0, tspan; kwargs...)
    return PSPMIPMProblem(SimpleIPM(), domain, n0, tspan; kwargs...)
end

function remake(prob::PSPMIPMProblem;
        structure = prob.structure,
        domain = prob.domain,
        n0 = prob.n0,
        aux0 = prob.aux0,
        tspan = prob.tspan,
        velocity = prob.velocity,
        mortality = prob.mortality,
        boundary_lower = prob.boundary_lower,
        boundary_upper = prob.boundary_upper,
        source = prob.source,
        auxiliary_rhs = prob.auxiliary_rhs,
        discretization = prob.discretization,
        p = prob.p,
        normalize = prob.normalize)
    return PSPMIPMProblem(structure, domain, n0, tspan;
        velocity = velocity,
        mortality = mortality,
        boundary_lower = boundary_lower,
        boundary_upper = boundary_upper,
        source = source,
        aux0 = aux0,
        auxiliary_rhs = auxiliary_rhs,
        discretization = discretization,
        p = p,
        normalize = normalize)
end

function Base.show(io::IO, prob::PSPMIPMProblem)
    print(io,
        "PSPMIPMProblem(",
        typeof(prob.structure).name.name,
        ", ",
        typeof(prob.discretization).name.name,
        ", tspan=",
        prob.tspan,
        ")")
end
