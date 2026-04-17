import OrdinaryDiffEq

using Serialization

_solution_matrix(sol) = reduce(hcat, sol.u)

const PBDM_PROJECT = normpath(joinpath(@__DIR__, "..", "..", "PhysiologicallyBasedDemographicModels.jl"))

function _run_pbdm_reference(script::AbstractString)
    mktemp() do path, io
        close(io)
        cmd = `$(Base.julia_cmd()) --project=$(PBDM_PROJECT) -e $script $path`
        run(cmd)
        open(path, "r") do ref_io
            return deserialize(ref_io)
        end
    end
end

function _continuous_reference()
    return _run_pbdm_reference("""
using Serialization
import OrdinaryDiffEq
import PhysiologicallyBasedDemographicModels
const PBDM = PhysiologicallyBasedDemographicModels

dev = PBDM.LinearDevelopmentRate(10.0, 35.0)
resource = PBDM.ContinuousSpecies(:plant;
    k = [3],
    τ = [100.0],
    μ = [0.0],
    dev_rate = [dev],
    fr = PBDM.FraserGilbertResponse(0.8),
    resp = PBDM.Q10Respiration(0.005, 2.0, 25.0),
    demand_rate = 0.5,
    intrinsic_rate = 0.5,
    carrying_capacity = 1000.0,
)
prob = PBDM.ContinuousPBDMProblem(
    species = [resource],
    u0 = [100.0, 100.0, 100.0],
    tspan = (0.0, 40.0),
    T_forcing = 25.0,
)
sol = PBDM.solve_continuous(
    prob;
    alg = OrdinaryDiffEq.Tsit5(),
    saveat = 1.0,
    reltol = 1e-8,
    abstol = 1e-10,
)
open(ARGS[1], "w") do io
    serialize(io, (t = sol.t, u = sol.u))
end
""")
end

function _pspm_reference()
    return _run_pbdm_reference("""
using Serialization
import OrdinaryDiffEq
import PhysiologicallyBasedDemographicModels
const PBDM = PhysiologicallyBasedDemographicModels

growth = 0.2
mortality = 0.01
fecundity(x, E, t) = x > 1.5 ? 0.05 * x : 0.0
init_density(x) = x < 2.0 ? 10.0 : 0.0

species = PBDM.PSPMSpecies(:daphnia;
    x_birth = 0.5,
    x_max = 5.0,
    growth_rate = (x, E, t) -> growth,
    mortality_rate = (x, E, t) -> mortality,
    fecundity_rate = fecundity,
    init_density = init_density,
)
prob = PBDM.PSPMProblem(
    species = [species],
    method = PBDM.FixedMeshUpwind(n_mesh = 40),
    tspan = (0.0, 20.0),
)
sol = PBDM.solve_pspm(
    prob;
    alg = OrdinaryDiffEq.Tsit5(),
    saveat = 1.0,
    reltol = 1e-8,
    abstol = 1e-10,
)
open(ARGS[1], "w") do io
    serialize(io, (t = sol.t, u = sol.u))
end
""")
end

@testset "PBDM pilot migrations" begin
    @testset "Exact continuous generator migration" begin
        ref = _continuous_reference()

        k = 3
        τ = 100.0
        μ = 0.0
        temperature = 25.0
        flow = (k / τ) * max(0.0, temperature - 10.0)
        respiration = 0.005

        function generator(u, p, t)
            G = zeros(Float64, k, k)
            for idx in 1:k
                G[idx, idx] = -(flow + μ * max(0.0, temperature - 10.0) + respiration)
                idx > 1 && (G[idx, idx - 1] = flow)
            end
            return G
        end

        function source(u, p, t)
            total = sum(u)
            result = zeros(Float64, k)
            result[1] = max(0.0, 0.5 * total * (1 - total / 1000.0))
            return result
        end

        prob = ContinuousIPMProblem(
            generator,
            ContinuousDomain(0.0, 1.0, 3),
            [100.0, 100.0, 100.0],
            (0.0, 40.0);
            source = source,
        )
        sol = solve(
            prob,
            OrdinaryDiffEq.Tsit5();
            saveat = 1.0,
            reltol = 1e-8,
            abstol = 1e-10,
        )

        migrated = _solution_matrix(sol)
        @test collect(sol.t) ≈ ref.t atol=1e-12
        @test migrated ≈ ref.u rtol=1e-6 atol=1e-6
        @test vec(sum(migrated; dims=1)) ≈ vec(sum(ref.u; dims=1)) rtol=1e-6 atol=1e-6
    end

    @testset "Exact PSPM fixed-mesh migration" begin
        ref = _pspm_reference()
        growth = 0.2
        mortality = 0.01
        fecundity(x, E, t) = x > 1.5 ? 0.05 * x : 0.0
        init_density(x) = x < 2.0 ? 10.0 : 0.0

        domain = ContinuousDomain(0.5, 5.0, 40)
        z = meshpoints(domain)
        dx = step_size(domain)
        n0 = init_density.(z)
        birth_flux(population, aux, p, t, domain) =
            sum(fecundity(x, nothing, t) * population[idx] * dx for (idx, x) in enumerate(z))

        prob = PSPMIPMProblem(
            domain,
            n0,
            (0.0, 20.0);
            velocity = growth,
            mortality = mortality,
            boundary_lower = birth_flux,
        )
        sol = solve(
            prob,
            OrdinaryDiffEq.Tsit5();
            saveat = 1.0,
            reltol = 1e-8,
            abstol = 1e-10,
        )

        migrated = _solution_matrix(sol)
        @test collect(sol.t) ≈ ref.t atol=1e-12
        @test migrated ≈ ref.u rtol=1e-6 atol=1e-6
    end
end
