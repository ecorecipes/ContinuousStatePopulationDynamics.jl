"""
Continuous-time generator problem types for continuous-state population
dynamics.
"""

"""
    AbstractContinuousIPMProblem

Supertype for continuous-state continuous-time formulations lowered to SciML
ODE/DDE problems.
"""
abstract type AbstractContinuousIPMProblem end
const AbstractContinuousStateDynamicsProblem = AbstractContinuousIPMProblem

"""
    ContinuousIPMProblem(structure, generator, domain, u0, tspan; p, source, normalize)

Continuous-time dynamics driven by an infinitesimal generator. The generator may
be a constant matrix or a callable returning one.
"""
struct ContinuousIPMProblem{
        S<:AbstractIPMStructure,
        G,
        Dom,
        U,
        T<:Real,
        P,
        B} <: AbstractContinuousIPMProblem
    structure::S
    generator::G
    domain::Dom
    u0::U
    tspan::Tuple{T, T}
    p::P
    source::B
    normalize::Bool
end

"""
    DelayIPMProblem(structure, generator, delay_terms, domain, u0, history, tspan;
                    p, source, normalize)

Delay problem with an instantaneous generator plus delayed linear
contributions.
"""
struct DelayIPMProblem{
        S<:AbstractIPMStructure,
        G,
        D,
        Dom,
        U,
        H,
        T<:Real,
        P,
        B} <: AbstractContinuousIPMProblem
    structure::S
    generator::G
    delay_terms::Vector{D}
    domain::Dom
    u0::U
    history::H
    tspan::Tuple{T, T}
    p::P
    source::B
    normalize::Bool
end

function _continuous_dimension(domain)
    if applicable(n_states, domain)
        return n_states(domain)
    elseif domain isa AbstractDict
        return sum(n_states(d) for d in values(domain))
    else
        return nothing
    end
end

function _validate_continuous_state(domain, u0)
    dim = _continuous_dimension(domain)
    dim === nothing && return
    length(u0) == dim || throw(DimensionMismatch(
        "initial state length $(length(u0)) does not match domain size $(dim)"))
end

function _continuous_numeric_type(u0, tspan)
    promote_type(Float64, eltype(u0), typeof(float(tspan[1])), typeof(float(tspan[2])))
end

function ContinuousIPMProblem(structure::AbstractIPMStructure, generator, domain, u0, tspan;
        p = nothing, source = nothing, normalize = false)
    u0_vec = collect(u0)
    isempty(u0_vec) && throw(ArgumentError("u0 must contain at least one state value"))
    _validate_continuous_state(domain, u0_vec)
    T = _continuous_numeric_type(u0_vec, tspan)
    return ContinuousIPMProblem(structure, generator, domain, T.(u0_vec),
        (T(tspan[1]), T(tspan[2])), p, source, normalize)
end

function ContinuousIPMProblem(generator, domain, u0, tspan; kwargs...)
    ContinuousIPMProblem(SimpleIPM(), generator, domain, u0, tspan; kwargs...)
end

function DelayIPMProblem(structure::AbstractIPMStructure, generator,
        delay_terms::AbstractVector{<:DelayGeneratorTerm},
        domain, u0, history, tspan;
        p = nothing, source = nothing, normalize = false)
    u0_vec = collect(u0)
    isempty(u0_vec) && throw(ArgumentError("u0 must contain at least one state value"))
    _validate_continuous_state(domain, u0_vec)
    T = _continuous_numeric_type(u0_vec, tspan)
    converted_terms = [DelayGeneratorTerm(T(term.lag), term.operator) for term in delay_terms]
    return DelayIPMProblem(structure, generator, converted_terms, domain, T.(u0_vec),
        history, (T(tspan[1]), T(tspan[2])), p, source, normalize)
end

function DelayIPMProblem(generator, delay_terms, domain, u0, history, tspan; kwargs...)
    DelayIPMProblem(SimpleIPM(), generator, delay_terms, domain, u0, history, tspan; kwargs...)
end

function remake(prob::ContinuousIPMProblem;
        structure = prob.structure,
        generator = prob.generator,
        domain = prob.domain,
        u0 = prob.u0,
        tspan = prob.tspan,
        p = prob.p,
        source = prob.source,
        normalize = prob.normalize)
    ContinuousIPMProblem(structure, generator, domain, u0, tspan;
        p = p, source = source, normalize = normalize)
end

function remake(prob::DelayIPMProblem;
        structure = prob.structure,
        generator = prob.generator,
        delay_terms = prob.delay_terms,
        domain = prob.domain,
        u0 = prob.u0,
        history = prob.history,
        tspan = prob.tspan,
        p = prob.p,
        source = prob.source,
        normalize = prob.normalize)
    DelayIPMProblem(structure, generator, delay_terms, domain, u0, history, tspan;
        p = p, source = source, normalize = normalize)
end

function Base.show(io::IO, prob::ContinuousIPMProblem)
    print(io, "ContinuousIPMProblem(", typeof(prob.structure).name.name,
        ", tspan=", prob.tspan, ")")
end

function Base.show(io::IO, prob::DelayIPMProblem)
    print(io, "DelayIPMProblem(", typeof(prob.structure).name.name,
        ", ", length(prob.delay_terms), " delays, tspan=", prob.tspan, ")")
end
