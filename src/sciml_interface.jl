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

function _materialize_generator_value(generator, u, p, t)
    if generator isa AbstractMatrix
        return generator
    elseif applicable(generator, u, p, t)
        return generator(u, p, t)
    elseif applicable(generator, u, t, p)
        return generator(u, t, p)
    elseif applicable(generator, p, t)
        return generator(p, t)
    elseif applicable(generator, t, p)
        return generator(t, p)
    else
        throw(ArgumentError(
            "generator $(typeof(generator)) is neither a matrix nor callable as (u, p, t)"))
    end
end

function _source_vector(source, u, p, t)
    source === nothing && return zeros(eltype(u), length(u))
    value = if source isa AbstractVector
        source
    elseif applicable(source, u, p, t)
        source(u, p, t)
    elseif applicable(source, u, t, p)
        source(u, t, p)
    elseif applicable(source, p, t)
        source(p, t)
    elseif applicable(source, t, p)
        source(t, p)
    else
        throw(ArgumentError(
            "source $(typeof(source)) is neither a vector nor callable as (u, p, t)"))
    end
    length(value) == length(u) || throw(DimensionMismatch(
        "source has length $(length(value)); expected $(length(u))"))
    return collect(eltype(u).(value))
end

function _lagged_state(history, p, t, n::Int, T::Type)
    value = history(p, t)
    length(value) == n || throw(DimensionMismatch(
        "history returned length $(length(value)); expected $(n)"))
    return collect(T.(value))
end

function _delay_term_matrix(term::DelayGeneratorTerm, u, history, p, t)
    op = term.operator
    value = if op isa AbstractMatrix
        op
    elseif applicable(op, u, history, p, t, term.lag)
        op(u, history, p, t, term.lag)
    elseif applicable(op, u, history, p, t)
        op(u, history, p, t)
    elseif applicable(op, u, p, t, term.lag)
        op(u, p, t, term.lag)
    elseif applicable(op, u, p, t)
        op(u, p, t)
    elseif applicable(op, p, t, term.lag)
        op(p, t, term.lag)
    elseif applicable(op, p, t)
        op(p, t)
    else
        throw(ArgumentError(
            "delay operator $(typeof(op)) is neither a matrix nor callable with supported signatures"))
    end
    return value
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

function _evaluate_field_value(field, points, population, aux, p, t; field_name::AbstractString)
    field === nothing && return nothing
    if field isa Number || field isa AbstractVector || field isa Tuple
        return field
    elseif applicable(field, points, population, aux, p, t)
        return field(points, population, aux, p, t)
    elseif applicable(field, points, population, p, t)
        return field(points, population, p, t)
    elseif applicable(field, points, aux, p, t)
        return field(points, aux, p, t)
    elseif applicable(field, points, p, t)
        return field(points, p, t)
    elseif applicable(field, points, t, p)
        return field(points, t, p)
    elseif applicable(field, points)
        return field(points)
    elseif !isempty(points)
        z = first(points)
        if applicable(field, z, population, aux, p, t)
            return [field(x, population, aux, p, t) for x in points]
        elseif applicable(field, z, population, p, t)
            return [field(x, population, p, t) for x in points]
        elseif applicable(field, z, aux, p, t)
            return [field(x, aux, p, t) for x in points]
        elseif applicable(field, z, p, t)
            return [field(x, p, t) for x in points]
        elseif applicable(field, z, t, p)
            return [field(x, t, p) for x in points]
        elseif applicable(field, z)
            return [field(x) for x in points]
        end
    end
    throw(ArgumentError(
        "$(field_name) $(typeof(field)) is not callable with a supported spatial signature"))
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

function _evaluate_spatial_field(field, points, population, aux, p, t, T::Type;
        field_name::AbstractString)
    value = _evaluate_field_value(field, points, population, aux, p, t;
        field_name = field_name)
    value === nothing && return zeros(T, length(points))
    return _coerce_field_values(value, length(points), T; field_name = field_name)
end

function _evaluate_face_field(field, faces, population, aux, p, t, T::Type;
        field_name::AbstractString)
    value = _evaluate_field_value(field, faces, population, aux, p, t;
        field_name = field_name)
    value === nothing && return zeros(T, length(faces))
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

function _evaluate_boundary_flux(flux, population, aux, p, t, domain, T::Type;
        field_name::AbstractString)
    if flux === nothing
        return zero(T)
    elseif flux isa Number
        return T(flux)
    elseif applicable(flux, population, aux, p, t, domain)
        return T(flux(population, aux, p, t, domain))
    elseif applicable(flux, population, aux, p, t)
        return T(flux(population, aux, p, t))
    elseif applicable(flux, population, p, t, domain)
        return T(flux(population, p, t, domain))
    elseif applicable(flux, population, p, t)
        return T(flux(population, p, t))
    elseif applicable(flux, aux, p, t, domain)
        return T(flux(aux, p, t, domain))
    elseif applicable(flux, aux, p, t)
        return T(flux(aux, p, t))
    elseif applicable(flux, p, t, domain)
        return T(flux(p, t, domain))
    elseif applicable(flux, p, t)
        return T(flux(p, t))
    elseif applicable(flux, t, p)
        return T(flux(t, p))
    end
    throw(ArgumentError(
        "$(field_name) $(typeof(flux)) is not callable with a supported boundary-flux signature"))
end

function _evaluate_auxiliary_rhs(rhs, population, aux, p, t, domain, T::Type)
    isempty(aux) && return T[]
    if rhs === nothing
        return zeros(T, length(aux))
    end

    value = if applicable(rhs, population, aux, p, t, domain)
        rhs(population, aux, p, t, domain)
    elseif applicable(rhs, population, aux, p, t)
        rhs(population, aux, p, t)
    elseif applicable(rhs, population, p, t, domain)
        rhs(population, p, t, domain)
    elseif applicable(rhs, population, p, t)
        rhs(population, p, t)
    else
        du_aux = zeros(T, length(aux))
        if applicable(rhs, du_aux, population, aux, p, t, domain)
            rhs(du_aux, population, aux, p, t, domain)
            return du_aux
        elseif applicable(rhs, du_aux, population, p, t, domain)
            rhs(du_aux, population, p, t, domain)
            return du_aux
        end
        throw(ArgumentError(
            "auxiliary_rhs $(typeof(rhs)) is not callable with a supported auxiliary-state signature"))
    end
    return _coerce_field_values(value, length(aux), T; field_name = "auxiliary_rhs")
end

"""
    to_ode_problem(prob::ContinuousIPMProblem)

Convert a continuous generator problem to a `SciMLBase.ODEProblem`.
"""
function to_ode_problem(prob::ContinuousIPMProblem)
    function ipm_ode!(du, u, p, t)
        G = _materialize_generator_matrix(_materialize_generator_value(prob.generator, u, p, t), length(u))
        mul!(du, G, u)
        du .+= _source_vector(prob.source, u, p, t)
        prob.normalize && _apply_continuous_normalization!(du, u)
        return nothing
    end
    return SciMLBase.ODEProblem(ipm_ode!, prob.u0, prob.tspan, prob.p)
end

"""
    to_dde_problem(prob::DelayIPMProblem)

Convert a delay generator problem to a `SciMLBase.DDEProblem`.
"""
function to_dde_problem(prob::DelayIPMProblem)
    lags = [term.lag for term in prob.delay_terms]
    function ipm_dde!(du, u, h, p, t)
        G = _materialize_generator_matrix(_materialize_generator_value(prob.generator, u, p, t), length(u))
        mul!(du, G, u)
        for term in prob.delay_terms
            lagged_u = _lagged_state(h, p, t - term.lag, length(u), eltype(u))
            delayed = _delay_term_matrix(term, u, h, p, t)
            if delayed isa AbstractVector
                length(delayed) == length(u) || throw(DimensionMismatch(
                    "delay term vector has length $(length(delayed)); expected $(length(u))"))
                du .+= eltype(u).(delayed)
            else
                Aτ = _materialize_generator_matrix(delayed, length(u))
                du .+= Aτ * lagged_u
            end
        end
        du .+= _source_vector(prob.source, u, p, t)
        prob.normalize && _apply_continuous_normalization!(du, u)
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

    function pspm_ode!(du, u, p, t)
        population = @view u[1:n_pop]
        du_population = @view du[1:n_pop]
        aux = @view u[(n_pop + 1):length(u)]
        du_aux = @view du[(n_pop + 1):length(u)]
        T = eltype(u)

        velocity = _evaluate_face_field(prob.velocity, face_points, population, aux, p, t, T;
            field_name = "velocity")
        mortality = _evaluate_spatial_field(prob.mortality, z, population, aux, p, t, T;
            field_name = "mortality")
        source = _evaluate_spatial_field(prob.source, z, population, aux, p, t, T;
            field_name = "source")
        lower_flux = _evaluate_boundary_flux(prob.boundary_lower, population, aux, p, t,
            prob.domain, T; field_name = "boundary_lower")
        upper_flux = _evaluate_boundary_flux(prob.boundary_upper, population, aux, p, t,
            prob.domain, T; field_name = "boundary_upper")

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
        prob.normalize && _apply_continuous_normalization!(du_population, population)

        if !isempty(aux)
            du_aux .= _evaluate_auxiliary_rhs(prob.auxiliary_rhs, population, aux, p, t,
                prob.domain, T)
        end
        return nothing
    end

    return SciMLBase.ODEProblem(pspm_ode!, u0, prob.tspan, prob.p)
end
