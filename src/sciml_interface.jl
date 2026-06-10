"""
SciML ecosystem interface for continuous-state continuous-time models.
"""

function generator_matrix(value::AbstractMatrix)
    return value
end

function generator_matrix(value)
    throw(ArgumentError(
        "generator must materialize to an AbstractMatrix, got $(typeof(value))"))
end

function _materialize_generator_matrix(value, n::Int)
    matrix = generator_matrix(value)
    size(matrix) == (n, n) || throw(DimensionMismatch(
        "generator matrix has size $(size(matrix)); expected ($n, $n)"))
    return matrix
end

# A calling-convention resolver. The argument convention of a user-supplied
# generator/source/operator is probed once (via `applicable`) when the SciML
# problem is lowered, and recorded as an integer `mode`. The hot RHS path then
# dispatches on that integer instead of re-running the `applicable` cascade on
# every function evaluation. `Resolved{F}` is concretely typed for a given `F`,
# so the caller captured by the lowered RHS keeps it type-stable.
struct Resolved{F}
    f::F
    mode::Int
end

# Shared convention for generator- and source-style callables.
# mode 0: constant value (matrix or vector); 1: (u,p,t); 2: (u,t,p); 3: (p,t); 4: (t,p).
@inline _call(r::Resolved, u, p, t) =
    r.mode == 0 ? r.f :
    r.mode == 1 ? r.f(u, p, t) :
    r.mode == 2 ? r.f(u, t, p) :
    r.mode == 3 ? r.f(p, t) : r.f(t, p)

function _resolve_generator(generator, u0, p, t0)
    generator isa AbstractMatrix && return Resolved(generator, 0)
    applicable(generator, u0, p, t0) && return Resolved(generator, 1)
    applicable(generator, u0, t0, p) && return Resolved(generator, 2)
    applicable(generator, p, t0) && return Resolved(generator, 3)
    applicable(generator, t0, p) && return Resolved(generator, 4)
    throw(ArgumentError(
        "generator $(typeof(generator)) is neither a matrix nor callable as (u, p, t)"))
end

function _resolve_source(source, u0, p, t0)
    source === nothing && return nothing
    source isa AbstractVector && return Resolved(source, 0)
    applicable(source, u0, p, t0) && return Resolved(source, 1)
    applicable(source, u0, t0, p) && return Resolved(source, 2)
    applicable(source, p, t0) && return Resolved(source, 3)
    applicable(source, t0, p) && return Resolved(source, 4)
    throw(ArgumentError(
        "source $(typeof(source)) is neither a vector nor callable as (u, p, t)"))
end

@inline _add_source!(du, ::Nothing, u, p, t) = nothing

@inline function _add_source!(du, source::Resolved, u, p, t)
    value = _call(source, u, p, t)
    length(value) == length(u) || throw(DimensionMismatch(
        "source has length $(length(value)); expected $(length(u))"))
    du .+= value
    return nothing
end

function _lagged_state(history, p, t, n::Int, T::Type)
    value = history(p, t)
    length(value) == n || throw(DimensionMismatch(
        "history returned length $(length(value)); expected $(n)"))
    return collect(T.(value))
end

# mode 0: constant matrix; 1: (u,h,p,t,lag); 2: (u,h,p,t); 3: (u,p,t,lag);
# 4: (u,p,t); 5: (p,t,lag); 6: (p,t).
@inline _call_delay(r::Resolved, u, h, p, t, lag) =
    r.mode == 0 ? r.f :
    r.mode == 1 ? r.f(u, h, p, t, lag) :
    r.mode == 2 ? r.f(u, h, p, t) :
    r.mode == 3 ? r.f(u, p, t, lag) :
    r.mode == 4 ? r.f(u, p, t) :
    r.mode == 5 ? r.f(p, t, lag) : r.f(p, t)

# Probing uses the user-supplied history function as a representative `h`. The
# runtime history interpolant has a different concrete type, but delay operators
# are written generically over `h`, so the resolved convention still matches.
function _resolve_delay_operator(op, u0, h0, p, t0, lag)
    op isa AbstractMatrix && return Resolved(op, 0)
    applicable(op, u0, h0, p, t0, lag) && return Resolved(op, 1)
    applicable(op, u0, h0, p, t0) && return Resolved(op, 2)
    applicable(op, u0, p, t0, lag) && return Resolved(op, 3)
    applicable(op, u0, p, t0) && return Resolved(op, 4)
    applicable(op, p, t0, lag) && return Resolved(op, 5)
    applicable(op, p, t0) && return Resolved(op, 6)
    throw(ArgumentError(
        "delay operator $(typeof(op)) is neither a matrix nor callable with supported signatures"))
end

function _apply_continuous_normalization!(du, u)
    total = sum(u)
    total > zero(total) || return
    growth = sum(du) / total
    du .-= growth .* u
end

function _coerce_field_values(value, n::Int, T::Type; field_name::AbstractString)
    if value isa Number
        return fill(T(value), n)
    end
    vec = collect(value)
    length(vec) == n || throw(DimensionMismatch(
        "$(field_name) has length $(length(vec)); expected $(n)"))
    return T.(vec)
end

# Spatial-field calling convention, resolved once against representative mesh
# points. Modes 1-6 are vectorized (the callable receives the whole points
# vector); modes 7-12 are the per-location fallback (the callable receives one
# point at a time and is broadcast over the mesh).
function _resolve_field(field, points0, population0, aux0, p, t0; field_name::AbstractString)
    field === nothing && return nothing
    (field isa Number || field isa AbstractVector || field isa Tuple) && return Resolved(field, 0)
    applicable(field, points0, population0, aux0, p, t0) && return Resolved(field, 1)
    applicable(field, points0, population0, p, t0) && return Resolved(field, 2)
    applicable(field, points0, aux0, p, t0) && return Resolved(field, 3)
    applicable(field, points0, p, t0) && return Resolved(field, 4)
    applicable(field, points0, t0, p) && return Resolved(field, 5)
    applicable(field, points0) && return Resolved(field, 6)
    if !isempty(points0)
        z = first(points0)
        applicable(field, z, population0, aux0, p, t0) && return Resolved(field, 7)
        applicable(field, z, population0, p, t0) && return Resolved(field, 8)
        applicable(field, z, aux0, p, t0) && return Resolved(field, 9)
        applicable(field, z, p, t0) && return Resolved(field, 10)
        applicable(field, z, t0, p) && return Resolved(field, 11)
        applicable(field, z) && return Resolved(field, 12)
    end
    throw(ArgumentError(
        "$(field_name) $(typeof(field)) is not callable with a supported spatial signature"))
end

@inline function _call_field(r::Resolved, points, population, aux, p, t)
    m = r.mode
    m == 0 && return r.f
    m == 1 && return r.f(points, population, aux, p, t)
    m == 2 && return r.f(points, population, p, t)
    m == 3 && return r.f(points, aux, p, t)
    m == 4 && return r.f(points, p, t)
    m == 5 && return r.f(points, t, p)
    m == 6 && return r.f(points)
    m == 7 && return [r.f(x, population, aux, p, t) for x in points]
    m == 8 && return [r.f(x, population, p, t) for x in points]
    m == 9 && return [r.f(x, aux, p, t) for x in points]
    m == 10 && return [r.f(x, p, t) for x in points]
    m == 11 && return [r.f(x, t, p) for x in points]
    return [r.f(x) for x in points]   # m == 12
end

function _midpoint_to_face(values)
    n = length(values)
    faces = Vector{eltype(values)}(undef, n + 1)
    faces[1] = values[1]
    for i in 1:(n - 1)
        faces[i + 1] = (values[i] + values[i + 1]) / 2
    end
    faces[end] = values[end]
    return faces
end

_spatial_values(::Nothing, points, population, aux, p, t, T::Type; field_name) =
    zeros(T, length(points))

function _spatial_values(r::Resolved, points, population, aux, p, t, T::Type;
        field_name::AbstractString)
    value = _call_field(r, points, population, aux, p, t)
    return _coerce_field_values(value, length(points), T; field_name = field_name)
end

_face_values(::Nothing, faces, population, aux, p, t, T::Type; field_name) =
    zeros(T, length(faces))

function _face_values(r::Resolved, faces, population, aux, p, t, T::Type;
        field_name::AbstractString)
    value = _call_field(r, faces, population, aux, p, t)
    if value isa Number
        return fill(T(value), length(faces))
    end
    vec = collect(value)
    if length(vec) == length(faces)
        return T.(vec)
    elseif length(vec) == length(faces) - 1
        return _midpoint_to_face(T.(vec))
    end
    throw(DimensionMismatch(
        "$(field_name) has length $(length(vec)); expected $(length(faces)) face values or $(length(faces) - 1) cell values"))
end

function _resolve_boundary_flux(flux, population0, aux0, p, t0, domain;
        field_name::AbstractString)
    flux === nothing && return nothing
    flux isa Number && return Resolved(flux, 0)
    applicable(flux, population0, aux0, p, t0, domain) && return Resolved(flux, 1)
    applicable(flux, population0, aux0, p, t0) && return Resolved(flux, 2)
    applicable(flux, population0, p, t0, domain) && return Resolved(flux, 3)
    applicable(flux, population0, p, t0) && return Resolved(flux, 4)
    applicable(flux, aux0, p, t0, domain) && return Resolved(flux, 5)
    applicable(flux, aux0, p, t0) && return Resolved(flux, 6)
    applicable(flux, p, t0, domain) && return Resolved(flux, 7)
    applicable(flux, p, t0) && return Resolved(flux, 8)
    applicable(flux, t0, p) && return Resolved(flux, 9)
    throw(ArgumentError(
        "$(field_name) $(typeof(flux)) is not callable with a supported boundary-flux signature"))
end

@inline _flux_value(::Nothing, population, aux, p, t, domain, T::Type) = zero(T)

@inline function _flux_value(r::Resolved, population, aux, p, t, domain, T::Type)
    m = r.mode
    m == 0 && return T(r.f)
    m == 1 && return T(r.f(population, aux, p, t, domain))
    m == 2 && return T(r.f(population, aux, p, t))
    m == 3 && return T(r.f(population, p, t, domain))
    m == 4 && return T(r.f(population, p, t))
    m == 5 && return T(r.f(aux, p, t, domain))
    m == 6 && return T(r.f(aux, p, t))
    m == 7 && return T(r.f(p, t, domain))
    m == 8 && return T(r.f(p, t))
    return T(r.f(t, p))   # m == 9
end

# Resolved once; returns `nothing` when there is no auxiliary state. Mode 0
# marks a missing rhs (zero derivative); modes 1-4 are out-of-place callables,
# modes 5-6 are in-place callables writing into a `du_aux` buffer.
function _resolve_aux_rhs(rhs, population0, aux0, p, t0, domain)
    isempty(aux0) && return nothing
    rhs === nothing && return Resolved(nothing, 0)
    applicable(rhs, population0, aux0, p, t0, domain) && return Resolved(rhs, 1)
    applicable(rhs, population0, aux0, p, t0) && return Resolved(rhs, 2)
    applicable(rhs, population0, p, t0, domain) && return Resolved(rhs, 3)
    applicable(rhs, population0, p, t0) && return Resolved(rhs, 4)
    du0 = zeros(eltype(aux0), length(aux0))
    applicable(rhs, du0, population0, aux0, p, t0, domain) && return Resolved(rhs, 5)
    applicable(rhs, du0, population0, p, t0, domain) && return Resolved(rhs, 6)
    throw(ArgumentError(
        "auxiliary_rhs $(typeof(rhs)) is not callable with a supported auxiliary-state signature"))
end

function _aux_du(r::Resolved, population, aux, p, t, domain, T::Type)
    m = r.mode
    m == 0 && return zeros(T, length(aux))
    m == 1 && return _coerce_field_values(r.f(population, aux, p, t, domain), length(aux), T;
        field_name = "auxiliary_rhs")
    m == 2 && return _coerce_field_values(r.f(population, aux, p, t), length(aux), T;
        field_name = "auxiliary_rhs")
    m == 3 && return _coerce_field_values(r.f(population, p, t, domain), length(aux), T;
        field_name = "auxiliary_rhs")
    m == 4 && return _coerce_field_values(r.f(population, p, t), length(aux), T;
        field_name = "auxiliary_rhs")
    du_aux = zeros(T, length(aux))
    if m == 5
        r.f(du_aux, population, aux, p, t, domain)
    else   # m == 6
        r.f(du_aux, population, p, t, domain)
    end
    return du_aux
end

"""
    to_ode_problem(prob::ContinuousIPMProblem)

Convert a continuous generator problem to a `SciMLBase.ODEProblem`.
"""
function to_ode_problem(prob::ContinuousIPMProblem)
    n = length(prob.u0)
    t0 = prob.tspan[1]
    gen = _resolve_generator(prob.generator, prob.u0, prob.p, t0)
    src = _resolve_source(prob.source, prob.u0, prob.p, t0)
    normalize = prob.normalize
    function ipm_ode!(du, u, p, t)
        G = _materialize_generator_matrix(_call(gen, u, p, t), n)
        mul!(du, G, u)
        _add_source!(du, src, u, p, t)
        normalize && _apply_continuous_normalization!(du, u)
        return nothing
    end
    return SciMLBase.ODEProblem(ipm_ode!, prob.u0, prob.tspan, prob.p)
end

"""
    to_dde_problem(prob::DelayIPMProblem)

Convert a delay generator problem to a `SciMLBase.DDEProblem`.
"""
function to_dde_problem(prob::DelayIPMProblem)
    n = length(prob.u0)
    t0 = prob.tspan[1]
    lags = [term.lag for term in prob.delay_terms]
    gen = _resolve_generator(prob.generator, prob.u0, prob.p, t0)
    src = _resolve_source(prob.source, prob.u0, prob.p, t0)
    delay_terms = prob.delay_terms
    resolved_ops = [_resolve_delay_operator(term.operator, prob.u0, prob.history, prob.p,
                                            t0, term.lag) for term in delay_terms]
    normalize = prob.normalize
    function ipm_dde!(du, u, h, p, t)
        G = _materialize_generator_matrix(_call(gen, u, p, t), n)
        mul!(du, G, u)
        for (term, rop) in zip(delay_terms, resolved_ops)
            delayed = _call_delay(rop, u, h, p, t, term.lag)
            if delayed isa AbstractVector
                length(delayed) == length(u) || throw(DimensionMismatch(
                    "delay term vector has length $(length(delayed)); expected $(length(u))"))
                du .+= eltype(u).(delayed)
            else
                Aτ = _materialize_generator_matrix(delayed, length(u))
                lagged_u = _lagged_state(h, p, t - term.lag, length(u), eltype(u))
                du .+= Aτ * lagged_u
            end
        end
        _add_source!(du, src, u, p, t)
        normalize && _apply_continuous_normalization!(du, u)
        return nothing
    end
    return SciMLBase.DDEProblem(ipm_dde!, prob.u0, prob.history, prob.tspan, prob.p;
        constant_lags = lags)
end

"""
    to_ode_problem(prob::PSPMIPMProblem)

Lower a fixed-mesh transport problem to a `SciMLBase.ODEProblem` using a
method-of-lines upwind discretization.
"""
function to_ode_problem(prob::PSPMIPMProblem)
    prob.discretization isa FixedMeshUpwind || throw(ArgumentError(
        "Only FixedMeshUpwind discretization is currently supported for PSPMIPMProblem"))

    z = meshpoints(prob.domain)
    face_points = bounds(prob.domain)
    h = step_size(prob.domain)
    n_pop = length(prob.n0)
    u0 = vcat(prob.n0, prob.aux0)
    domain = prob.domain
    normalize = prob.normalize
    t0 = prob.tspan[1]

    # Resolve each field's calling convention once, against representative
    # initial state, so the RHS does not re-probe `applicable` per step.
    rvel = _resolve_field(prob.velocity, face_points, prob.n0, prob.aux0, prob.p, t0;
        field_name = "velocity")
    rmort = _resolve_field(prob.mortality, z, prob.n0, prob.aux0, prob.p, t0;
        field_name = "mortality")
    rsrc = _resolve_field(prob.source, z, prob.n0, prob.aux0, prob.p, t0;
        field_name = "source")
    rlow = _resolve_boundary_flux(prob.boundary_lower, prob.n0, prob.aux0, prob.p, t0, domain;
        field_name = "boundary_lower")
    rup = _resolve_boundary_flux(prob.boundary_upper, prob.n0, prob.aux0, prob.p, t0, domain;
        field_name = "boundary_upper")
    raux = _resolve_aux_rhs(prob.auxiliary_rhs, prob.n0, prob.aux0, prob.p, t0, domain)

    function pspm_ode!(du, u, p, t)
        population = @view u[1:n_pop]
        du_population = @view du[1:n_pop]
        aux = @view u[(n_pop + 1):length(u)]
        du_aux = @view du[(n_pop + 1):length(u)]
        T = eltype(u)

        velocity = _face_values(rvel, face_points, population, aux, p, t, T;
            field_name = "velocity")
        mortality = _spatial_values(rmort, z, population, aux, p, t, T;
            field_name = "mortality")
        source = _spatial_values(rsrc, z, population, aux, p, t, T;
            field_name = "source")
        lower_flux = _flux_value(rlow, population, aux, p, t, domain, T)
        upper_flux = _flux_value(rup, population, aux, p, t, domain, T)

        fluxes = Vector{T}(undef, n_pop + 1)
        fluxes[1] = velocity[1] >= zero(T) ? lower_flux : velocity[1] * population[1]
        for i in 1:(n_pop - 1)
            vi = velocity[i + 1]
            fluxes[i + 1] = vi >= zero(T) ? vi * population[i] : vi * population[i + 1]
        end
        fluxes[end] = velocity[end] >= zero(T) ? velocity[end] * population[end] : upper_flux

        for i in 1:n_pop
            du_population[i] = -(fluxes[i + 1] - fluxes[i]) / h -
                mortality[i] * population[i] + source[i]
        end
        normalize && _apply_continuous_normalization!(du_population, population)

        if !isempty(aux)
            du_aux .= _aux_du(raux, population, aux, p, t, domain, T)
        end
        return nothing
    end

    return SciMLBase.ODEProblem(pspm_ode!, u0, prob.tspan, prob.p)
end
