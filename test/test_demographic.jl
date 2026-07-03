using Random
using LinearAlgebra
using StructuredPopulationCore: cle_drift!, cle_noise!, num_reactions, quasi_extinction
import SciMLBase
import OrdinaryDiffEq

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

        inflow_prob = PSPMIPMProblem(td, zeros(3), (0.0, 1.0);
            velocity=-0.5, mortality=0.0, source=0.0, boundary_upper=-3.0)
        inflow_noise = zeros(3)
        to_sde_problem(inflow_prob).f.g(inflow_noise, inflow_prob.n0, inflow_prob.p, 0.0)
        @test inflow_noise[end] ≈ sqrt(3.0 / step_size(td))
    end

    @testset "PSPM exact Gillespie: mean tracks deterministic transport" begin
        td = ContinuousDomain(0.0, 1.0, 5)
        n0 = [20, 20, 20, 20, 20]
        prob = PSPMIPMProblem(td, n0, (0.0, 1.0);
            velocity=0.5, mortality=0.1, boundary_lower=4.0)   # constant linear fields

        sol = OrdinaryDiffEq.solve(to_ode_problem(prob), OrdinaryDiffEq.Tsit5();
            saveat=0.25, reltol=1e-9, abstol=1e-11)
        grid = 0.0:0.25:1.0
        reps = 4000
        acc = [zeros(5) for _ in grid]
        for _ in 1:reps
            s = solve(prob, Demographic(); rng=rng, saveat=grid)
            for g in eachindex(grid)
                acc[g] .+= s.u[g]
            end
        end
        for g in eachindex(grid)
            @test isapprox(acc[g] ./ reps, sol.u[g]; rtol=0.08, atol=1.5)
        end

        # coupled auxiliary ODE state is not supported by the jump route
        paux = PSPMIPMProblem(td, n0, (0.0, 1.0); velocity=0.5, aux0=[1.0],
            auxiliary_rhs=(pop, a, p, t, d) -> [0.0])
        @test_throws ArgumentError solve(paux, Demographic())
    end

    @testset "exact solves validate integer initial counts and retcodes" begin
        cprob = ContinuousIPMProblem(zeros(2, 2), domain, [10.0, 0.0], (0.0, 0.01); source=[0.0, 0.0])
        csol = solve(cprob, Demographic(); rng=rng)
        @test csol.u[1] == [10, 0]
        @test_throws ArgumentError solve(
            ContinuousIPMProblem(zeros(2, 2), domain, [10.25, 0.0], (0.0, 0.01)),
            Demographic())

        pprob = PSPMIPMProblem(domain, [10.0, 0.0], (0.0, 0.01); velocity=0.0, source=0.0)
        psol = solve(pprob, Demographic(); rng=rng)
        @test psol.u[1] == [10, 0]
        @test_throws ArgumentError solve(
            PSPMIPMProblem(domain, [10.25, 0.0], (0.0, 0.01); velocity=0.0, source=0.0),
            Demographic())

        maxiters_prob = ContinuousIPMProblem(zeros(2, 2), domain, [0.0, 0.0], (0.0, 1.0);
            source=[1000.0, 0.0])
        @test solve(maxiters_prob, Demographic(); rng=rng, max_events=1).retcode == :MaxIters

        maxiters_pspm = PSPMIPMProblem(domain, [0.0, 0.0], (0.0, 1.0); velocity=0.0, source=1000.0)
        @test solve(maxiters_pspm, Demographic(); rng=rng, max_events=1).retcode == :MaxIters
    end

    @testset "PSPM exact solve rejects state/time-dependent callables" begin
        td = ContinuousDomain(0.0, 1.0, 3)
        static_prob = PSPMIPMProblem(td, [5, 5, 5], (0.0, 0.1);
            velocity=points -> fill(0.5, length(points)),
            mortality=points -> fill(0.1, length(points)),
            source=points -> zeros(length(points)))
        @test solve(static_prob, Demographic(); rng=rng).retcode == :Success

        bad_prob = PSPMIPMProblem(td, [5, 5, 5], (0.0, 0.1);
            velocity=(points, population, p, t) -> fill(0.5 + 0.01 * sum(population), length(points)))
        @test_throws ArgumentError solve(bad_prob, Demographic())
    end

    @testset "errors: callable generator is not demographic-realizable" begin
        cp = ContinuousIPMProblem((u, p, t) -> [-0.1 0.0; 0.0 -0.1], domain,
            [10.0, 10.0], (0.0, 1.0))
        @test_throws ArgumentError solve(cp, Demographic())
        @test_throws ArgumentError to_sde_problem(cp)
    end
end
