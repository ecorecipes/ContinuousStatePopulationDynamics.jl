using LinearAlgebra
using Random
using ResilienceCore
import ForwardDiff
import OrdinaryDiffEq

const CSPD = ContinuousStatePopulationDynamics

# central finite differences of an in-place right-hand side
function fd_jacobian(f!, u, p, t; h = 1e-6)
    n = length(u)
    J = zeros(n, n)
    up, dn = zeros(n), zeros(n)
    for j in 1:n
        d = h * max(abs(u[j]), 1.0)
        x = copy(u); x[j] += d; f!(up, x, p, t)
        x = copy(u); x[j] -= d; f!(dn, x, p, t)
        J[:, j] = (up .- dn) ./ (2d)
    end
    return J
end

# a continuous IPM generator on bin masses: births at size 0 from all sizes, growth by
# transfer to the next bin, size-dependent mortality
function ipm_generator(n; L = 1.0, growth = 0.6, birth = 1.2, death = z -> 0.3 + z)
    domain = ContinuousDomain(0.0, L, n)
    z = meshpoints(domain)
    h = step_size(domain)
    G = zeros(n, n)
    for i in 1:n
        G[i, i] -= death(z[i]) + growth / h
        i < n && (G[i + 1, i] += growth / h)
        G[1, i] += birth * exp(-z[i])
    end
    return G, domain
end

@testset "ResilienceCore operators" begin
    domain = ContinuousDomain(0.0, 1.0, 3)

    @testset "constant generator: Generator with weights" begin
        G = [-0.5 0.2 0.0; 0.1 -0.3 0.1; 0.0 0.2 -0.4]
        prob = ContinuousIPMProblem(G, domain, [1.0, 1.0, 1.0], (0.0, 1.0))
        L = linearise(prob)
        @test L isa Generator
        @test operator(L) == G
        @test growth_rate(L) ≈ maximum(real, eigvals(G))
        @test ResilienceCore.weights(L) == ones(3)
        Lw = linearise(prob; weights = mesh_weights(domain))
        @test ResilienceCore.weights(Lw) ≈ fill(step_size(domain), 3)
        # a common weight changes no norm-based metric
        @test reactivity(Lw) ≈ reactivity(L)
        # normalisation leaves the operator of the unnormalised problem
        @test operator(linearise(remake(prob; normalize = true))) == G
    end

    @testset "generator callable in (p, t) is evaluated at t" begin
        prob = ContinuousIPMProblem((p, t) -> [-p.a*(1 + t) 0.0; 0.0 -1.0], ContinuousDomain(0.0, 1.0, 2),
                                    [1.0, 1.0], (0.0, 1.0); p = (a = 0.5,))
        @test operator(linearise(prob)) ≈ [-0.5 0.0; 0.0 -1.0]
        @test operator(linearise(prob; t = 1.0)) ≈ [-1.0 0.0; 0.0 -1.0]
    end

    @testset "growth rate and resilience converge under mesh refinement" begin
        g = Float64[]
        r = Float64[]
        for n in (25, 50, 100, 200, 400)
            G, d = ipm_generator(n)
            L = linearise(ContinuousIPMProblem(G, d, ones(n), (0.0, 1.0)))
            push!(g, growth_rate(L))
            push!(r, resilience(L))
        end
        dg = abs.(diff(g))
        @test all(dg[2:end] .< dg[1:end-1])            # successive changes shrink
        @test dg[end] < 0.01
        dr = abs.(diff(r))                              # resilience: the gap Re λ₁ − Re λ₂
        @test all(dr[2:end] .< dr[1:end-1])
        @test dr[end] < 0.02 * abs(r[end])
    end

    @testset "input term: equilibrium and Jacobian" begin
        G = [-0.5 0.2 0.0; 0.1 -0.3 0.1; 0.0 0.2 -0.4]
        b = [0.3, 0.0, 0.1]
        prob = ContinuousIPMProblem(G, domain, ones(3), (0.0, 1.0); source = b)
        J = linearise(prob)
        @test J isa ContinuousJacobian
        @test J.J == G
        @test G * J.equilibrium .+ b ≈ zeros(3) atol = 1e-12
        # the equilibrium is the long-run state of the problem
        sol = OrdinaryDiffEq.solve(to_ode_problem(remake(prob; tspan = (0.0, 200.0))), OrdinaryDiffEq.Tsit5();
                                   abstol = 1e-12, reltol = 1e-10)
        @test sol.u[end] ≈ J.equilibrium rtol = 1e-6
        # an unstable generator has no equilibrium to return to
        @test_throws ArgumentError linearise(ContinuousIPMProblem([0.1 0.0; 0.0 -1.0], ContinuousDomain(0.0, 1.0, 2),
                                                                  [1.0, 1.0], (0.0, 1.0); source = [1.0, 1.0]))
        @test_throws ArgumentError linearise(remake(prob; normalize = true))
    end

    @testset "periodic generator" begin
        T = 2.0
        Q = [-0.6 0.3; 0.2 -0.5]
        f(t) = 1 + 0.5cos(2π * t / T)^2
        prob = ContinuousIPMProblem((p, t) -> f(t) .* Q, ContinuousDomain(0.0, 1.0, 2), [1.0, 1.0], (0.0, 10.0))
        P = linearise(prob; period = T, steps = 64)
        @test P isa PeriodicOperator
        @test period(P) == 64
        # commuting case: the period map is exp(Q ∫f)
        @test operator(monodromy(P)) ≈ exp(Q * 1.25T) rtol = 1e-10
        # non-commuting case: the midpoint product converges to the period map of the ODE
        Q2 = [0.3 0.2; 0.15 -0.2]                        # Q + s Q2 keeps non-negative off-diagonals for |s| ≤ 1
        @test norm(Q * Q2 - Q2 * Q) > 0.01
        prob2 = ContinuousIPMProblem((p, t) -> Q .+ sin(2π * t / T) .* Q2, ContinuousDomain(0.0, 1.0, 2),
                                     [1.0, 1.0], (0.0, 10.0))
        Φ = hcat([OrdinaryDiffEq.solve(OrdinaryDiffEq.ODEProblem((u, p, t) -> (Q .+ sin(2π * t / T) .* Q2) * u, e,
                                                                 (0.0, T)), OrdinaryDiffEq.Vern9();
                                       abstol = 1e-13, reltol = 1e-13).u[end] for e in ([1.0, 0.0], [0.0, 1.0])]...)
        err(k) = opnorm(operator(monodromy(linearise(prob2; period = T, steps = k))) .- Φ)
        @test err(200) < err(50) / 8                   # second order
        @test err(200) < 1e-4
        @test_throws ArgumentError linearise(remake(prob; source = [0.1, 0.0]); period = T)
    end

    @testset "state-dependent generator: the SciML route at an equilibrium" begin
        # logistic stage dynamics: dn/dt = G(n) n with crowding on recruitment
        K = 10.0
        gen(u, p, t) = [-0.2 0.8 * (1 - (u[1] + u[2]) / K); 0.5 -0.4]
        prob = ContinuousIPMProblem(gen, ContinuousDomain(0.0, 1.0, 2), [1.0, 1.0], (0.0, 1.0))
        @test_throws ArgumentError linearise(prob)
        f! = to_ode_problem(prob).f
        sol = OrdinaryDiffEq.solve(to_ode_problem(remake(prob; tspan = (0.0, 2000.0))), OrdinaryDiffEq.Tsit5();
                                   abstol = 1e-13, reltol = 1e-12)
        ustar = sol.u[end]
        J = linearise(prob, ustar)
        @test J isa ContinuousJacobian
        @test J.J ≈ fd_jacobian(f!, ustar, nothing, 0.0) rtol = 1e-6
        @test J.equilibrium ≈ ustar
        @test_throws ArgumentError linearise(prob, [1.0, 1.0])            # not an equilibrium
        @test linearise(prob, [1.0, 1.0]; check = false) isa ContinuousJacobian
    end

    @testset "PSPM transport: exact assembly" begin
        d = ContinuousDomain(0.0, 2.0, 40)
        # velocity changes sign inside the domain, so both upwind branches are used
        prob = PSPMIPMProblem(d, ones(40), (0.0, 1.0);
                              velocity = z -> 0.5 .- 0.4 .* z,
                              mortality = z -> 0.1 .+ 0.05 .* z,
                              source = z -> 0.2 .* exp.(-z),
                              boundary_lower = 0.3)
        f! = to_ode_problem(prob).f
        T, b = CSPD._affine_parts(f!, 40, nothing, 0.0)
        rng = Xoshiro(1)
        du = zeros(40)
        for _ in 1:5
            x = rand(rng, 40) .* 3
            f!(du, x, nothing, 0.0)
            @test du ≈ T * x .+ b rtol = 1e-12
        end
        J = linearise(prob)
        @test J isa ContinuousJacobian
        @test J.J ≈ T
        @test T * J.equilibrium .+ b ≈ zeros(40) atol = 1e-10
        # with no inflow and no source, a Generator with the bin widths as weights
        L = linearise(remake(prob; source = 0.0, boundary_lower = 0.0))
        @test L isa Generator
        @test ResilienceCore.weights(L) ≈ mesh_weights(d)
        @test operator(L) ≈ T
    end

    @testset "PSPM equilibrium converges to the stationary density" begin
        # v n' = -μ n with n(0) = B / v: n*(z) = (B / v) exp(-μ z / v)
        v, μ, B = 0.5, 0.4, 1.0
        errs = map((50, 100, 200, 400)) do n
            d = ContinuousDomain(0.0, 3.0, n)
            prob = PSPMIPMProblem(d, ones(n), (0.0, 1.0); velocity = v, mortality = μ, boundary_lower = B)
            J = linearise(prob)
            exact = (B / v) .* exp.(-μ .* meshpoints(d) ./ v)
            maximum(abs, J.equilibrium .- exact)
        end
        @test all(diff(collect(errs)) .< 0)
        @test errs[end] < 0.02
    end

    @testset "PSPM with a resource: the SciML route" begin
        d = ContinuousDomain(0.0, 1.0, 20)
        h = step_size(d)
        r, Kr, c, β, μ = 1.0, 2.0, 0.5, 0.8, 0.3
        prob = PSPMIPMProblem(d, fill(0.5, 20), (0.0, 1.0);
                              velocity = (z, n, aux, p, t) -> aux[1] .* (1.05 .- z),
                              mortality = μ,
                              boundary_lower = (n, aux, p, t) -> β * aux[1] * h * sum(n),
                              aux0 = [1.0],
                              auxiliary_rhs = (n, aux, p, t) -> [r * (Kr - aux[1]) - c * aux[1] * h * sum(n)])
        @test_throws ArgumentError linearise(prob)                       # auxiliary state needs `u`
        f! = to_ode_problem(prob).f
        u = vcat(fill(0.5, 20), [1.0])
        J = linearise(prob, u; check = false)
        @test J isa ContinuousJacobian
        @test J.J ≈ fd_jacobian(f!, u, nothing, 0.0) rtol = 1e-5 atol = 1e-7
        @test_throws ArgumentError linearise(prob, u)                    # not an equilibrium
        # density dependence without auxiliary state is detected
        dd = PSPMIPMProblem(d, fill(0.5, 20), (0.0, 1.0); velocity = 0.5,
                            mortality = (z, n, p, t) -> 0.1 .+ 0.2 .* h .* sum(n), boundary_lower = 0.2)
        @test_throws ArgumentError linearise(dd)
        @test_throws ArgumentError linearise(prob, u; period = 1.0)
    end

    @testset "problems without a single operator" begin
        G = [-0.5 0.0; 0.2 -0.1]
        delay = DelayGeneratorTerm(1.0, [0.0 0.4; 0.0 0.0])
        dprob = DelayIPMProblem(G, [delay], ContinuousDomain(0.0, 1.0, 2), [1.0, 0.5], (p, t) -> [2.0, 1.0],
                                (0.0, 2.0))
        @test_throws ArgumentError linearise(dprob)
        sprob = ContinuousIPMProblem([-0.8 0.4; 0.8 -0.4], ContinuousDomain(0.0, 1.0, 2), [10, 0], (0.0, 1.0))
        sol = solve(sprob, Demographic(); rng = Xoshiro(2))
        @test_throws ArgumentError linearise(sol)
    end

    @testset "no name clashes with ResilienceCore" begin
        shared = intersect(names(CSPD), names(ResilienceCore))
        @test :linearise in shared
        @test all(s -> getfield(CSPD, s) === getfield(ResilienceCore, s), shared)
    end
end
