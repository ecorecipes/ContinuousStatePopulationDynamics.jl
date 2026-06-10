using Random
using LinearAlgebra
using StructuredPopulationCore: cle_drift!, cle_noise!, num_reactions, quasi_extinction
import SciMLBase

# Hand-rolled Euler-Maruyama mean for a reaction-CLE (avoids a heavy SDE solver dep).
function _em_mean(sys, u0, tspan, dt, reps, rng)
    n = length(u0)
    m = num_reactions(sys)
    acc = zeros(n)
    du = zeros(n)
    M = zeros(n, m)
    nsteps = round(Int, (tspan[2] - tspan[1]) / dt)
    for _ in 1:reps
        u = collect(float.(u0))
        for _ in 1:nsteps
            cle_drift!(du, sys, u, nothing, 0.0)
            cle_noise!(M, sys, u, nothing, 0.0)
            u .+= du .* dt .+ M * (sqrt(dt) .* randn(rng, m))
        end
        acc .+= u
    end
    return acc ./ reps
end

@testset "Demographic stochasticity (continuous-state continuous-time)" begin
    rng = Random.Xoshiro(11)
    domain = ContinuousDomain(0.0, 1.0, 2)

    @testset "generator exact Gillespie: mean = exp(Gt)·n0, conserved" begin
        G = [-0.8 0.4; 0.8 -0.4]                       # conservative migration
        n0 = [100, 0]
        prob = ContinuousIPMProblem(G, domain, n0, (0.0, 2.0))
        grid = 0.0:0.5:2.0
        reps = 4000
        acc = [zeros(2) for _ in grid]
        conserved = true
        for _ in 1:reps
            s = solve(prob, Demographic(); rng=rng, saveat=grid)
            for g in eachindex(grid)
                acc[g] .+= s.u[g]
            end
            all(sum(u) == 100 for u in s.u) || (conserved = false)
        end
        @test conserved
        for (g, t) in enumerate(grid)
            @test isapprox(acc[g] ./ reps, exp(G .* t) * n0; rtol=0.05, atol=1.0)
        end
    end

    @testset "generator CLE: drift = G·u + source, valid diffusion" begin
        G = [-0.5 0.3; 0.5 -0.6]
        src = [1.0, 0.0]
        prob = ContinuousIPMProblem(G, domain, [10.0, 5.0], (0.0, 1.0); source=src)
        sys = to_demographic_reactions(prob)
        u = [10.0, 5.0]
        du = zeros(2); cle_drift!(du, sys, u, nothing, 0.0)
        @test isapprox(du, G * u .+ src; atol=1e-10)              # drift = deterministic RHS
        M = zeros(2, num_reactions(sys)); cle_noise!(M, sys, u, nothing, 0.0)
        D = M * M'
        @test isapprox(D, D'; atol=1e-12) && all(diag(D) .>= 0)   # valid diffusion

        sde = to_sde_problem(prob)
        @test sde isa SciMLBase.SDEProblem
        du2 = zeros(2); sde.f.f(du2, u, sde.p, 0.0)
        @test isapprox(du2, G * u .+ src; atol=1e-10)             # SDE drift = RHS
    end

    @testset "generator CLE: Euler-Maruyama mean tracks deterministic" begin
        G = [-0.5 0.3; 0.5 -0.6]
        n0 = [200.0, 100.0]
        prob = ContinuousIPMProblem(G, domain, n0, (0.0, 1.0))
        sys = to_demographic_reactions(prob)
        em = _em_mean(sys, n0, (0.0, 1.0), 0.002, 3000, rng)
        @test isapprox(em, exp(G .* 1.0) * n0; rtol=0.05, atol=2.0)
    end

    @testset "generator ensemble + quasi_extinction (subcritical)" begin
        G = [-0.6 0.1; 0.2 -0.7]                        # both eigenvalues < 0
        prob = ContinuousIPMProblem(G, domain, [30, 30], (0.0, 40.0))
        totals, _ = demographic_ensemble(prob; n_reps=300, saveat=1.0, rng=rng)
        @test size(totals, 1) == 41
        @test quasi_extinction(totals; threshold=1.0).prob_extinct > 0.5
    end

    @testset "PSPM CLE: drift = transport RHS, diagonal demographic noise" begin
        td = ContinuousDomain(0.0, 1.0, 3)
        n0 = [10.0, 8.0, 6.0]
        prob = PSPMIPMProblem(td, n0, (0.0, 1.0); velocity=0.5, mortality=0.2, source=0.1)
        sde = to_sde_problem(prob)
        @test sde isa SciMLBase.SDEProblem

        ode = to_ode_problem(prob)
        du_sde = zeros(3); du_ode = zeros(3)
        sde.f.f(du_sde, n0, sde.p, 0.0)
        ode.f(du_ode, n0, ode.p, 0.0)
        @test isapprox(du_sde, du_ode; atol=1e-12)                # drift = deterministic transport

        gv = zeros(3); sde.f.g(gv, n0, sde.p, 0.0)
        # advection is drift-only; noise = sqrt(mortality·n + source); boundaries are 0 here
        @test isapprox(gv, sqrt.(0.2 .* n0 .+ 0.1); atol=1e-10)
    end

    @testset "errors: callable generator is not demographic-realizable" begin
        cp = ContinuousIPMProblem((u, p, t) -> [-0.1 0.0; 0.0 -0.1], domain,
            [10.0, 10.0], (0.0, 1.0))
        @test_throws ArgumentError solve(cp, Demographic())
        @test_throws ArgumentError to_sde_problem(cp)
    end
end
